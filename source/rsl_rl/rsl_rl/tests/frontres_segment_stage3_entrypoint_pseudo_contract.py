#!/usr/bin/env python3
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "source" / "rsl_rl"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
TRAIN_PATH = ROOT / "scripts" / "rsl_rl" / "train.py"
MOSAIC_CFG_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "config"
    / "g1"
    / "agents"
    / "rsl_rl_mosaic_cfg.py"
)


def _load_stage_preset():
    rsl_rl = types.ModuleType("rsl_rl")
    rsl_rl.__path__ = [str(SOURCE_ROOT / "rsl_rl")]
    frontres = types.ModuleType("rsl_rl.frontres")
    frontres.__path__ = [str(SOURCE_ROOT / "rsl_rl" / "frontres")]
    sys.modules["rsl_rl"] = rsl_rl
    sys.modules["rsl_rl.frontres"] = frontres
    probe = types.ModuleType("rsl_rl.frontres.frontres_formal_runtime_probe")
    probe.emit_formal_runtime_probe = lambda *args, **kwargs: None
    sys.modules[probe.__name__] = probe
    warmup_path = ROOT / "source" / "rsl_rl" / "rsl_rl" / "frontres" / "frontres_segment_warmup.py"
    warmup_spec = importlib.util.spec_from_file_location("rsl_rl.frontres.frontres_segment_warmup", warmup_path)
    assert warmup_spec is not None and warmup_spec.loader is not None
    warmup_module = importlib.util.module_from_spec(warmup_spec)
    sys.modules[warmup_spec.name] = warmup_module
    warmup_spec.loader.exec_module(warmup_module)
    tree = ast.parse(TRAIN_PATH.read_text())
    wanted = {
        "_default_frontres_segment_cache_dir",
        "_set_if_present",
        "_parse_frontres_v015_future_offsets",
        "_parse_frontres_v015_k_curriculum",
        "_apply_frontres_stage_preset",
        "_apply_frontres_segment_ppo_schedule_override",
        "_apply_frontres_segment_ppo_lr_override",
        "_configure_frontres_stage3_segment_hrl_env_cfg",
    }
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "RslRlOnPolicyRunnerCfg": object,
        "os": os,
        "_FEMR_DATA_ROOT": str(ROOT.parent),
    }
    exec(compile(module, str(TRAIN_PATH), "exec"), namespace)
    return (
        namespace["_apply_frontres_stage_preset"],
        namespace["_apply_frontres_segment_ppo_schedule_override"],
        namespace["_apply_frontres_segment_ppo_lr_override"],
        namespace["_configure_frontres_stage3_segment_hrl_env_cfg"],
    )


(
    _apply_frontres_stage_preset,
    _apply_frontres_segment_ppo_schedule_override,
    _apply_frontres_segment_ppo_lr_override,
    _configure_frontres_stage3_segment_hrl_env_cfg,
) = _load_stage_preset()


def _alg_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        frontres_training_objective="unset",
        frontres_segment_replay_enabled=False,
        frontres_segment_live_runner_enabled=False,
        frontres_segment_live_sentinel_only=False,
        frontres_segment_live_probe_only=False,
        frontres_segment_live_storage_write_only=False,
        frontres_segment_live_single_update_only=False,
        frontres_segment_live_update_loop_only=False,
        frontres_segment_live_train_enabled=False,
        frontres_formal_transaction_enabled=False,
        frontres_segment_live_update_steps=4,
        frontres_segment_critic_warmup_iterations=0,
        frontres_segment_actor_warmup_iterations=0,
        frontres_segment_k_curriculum=(),
        frontres_formal_runtime_audit=False,
        frontres_hsl_init_enabled=False,
        frontres_hsl_rollout_label_enabled=True,
        lambda_supervised=1.0,
        lambda_supervised_min=0.05,
        frontres_segment_k=0,
        frontres_future_offsets=(),
        frontres_future_intent_layout_version="unset",
        frontres_segment_max_horizon_k=0,
        frontres_segment_advantage_normalization="standard",
        frontres_segment_sampler_global_frac=0.0,
        frontres_segment_sampler_replay_frac=0.0,
        frontres_segment_sampler_review_frac=0.0,
        frontres_segment_reset_mode="unset",
        schedule="fixed",
        learning_rate=1.0e-4,
    )


def _policy_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_task_corrections=6,
    )


def _agent_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        algorithm=_alg_cfg(),
        policy=_policy_cfg(),
        experiment_name="unset",
        is_full_resume=True,
        frontres_stage1_exit_after_warmup=True,
        supervised_warmup_iterations=99,
        critic_warmup_iterations=99,
        max_iterations=11,
        save_interval=100,
    )


def _args(**overrides) -> SimpleNamespace:
    values = {
        "frontres_stage": "stage3_segment_hrl",
        "frontres_segment_live_sentinel_only": False,
        "frontres_segment_live_probe_only": False,
        "frontres_segment_live_storage_write_only": False,
        "frontres_segment_live_single_update_only": False,
        "frontres_segment_live_update_loop_only": False,
        "frontres_segment_live_update_steps": 6,
        "frontres_segment_critic_warmup_iterations": 200,
        "frontres_segment_actor_warmup_iterations": 500,
        "frontres_segment_k_curriculum": "8:2:200:500:1300:lower-k8:0.5:linear-joint-v1:1300:2.381,16:3:300:300:900:lower-k16:0.6:linear-joint-v1:900:2.381,32:4:400:300:625:lower-k32:0.7:linear-joint-v1:625:2.381",
        "frontres_formal_runtime_audit": False,
        "frontres_v015_future_offsets": "1,2",
        "frontres_v015_hsl_initializer_checkpoint": "/tmp/frontres-v017-hsl-proposal-v2.pt",
        "frontres_segment_ppo_schedule": None,
        "frontres_segment_ppo_lr": None,
        "experiment_name": None,
        "is_full_resume": None,
        "frontres_checkpoint_interval": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _probe_stage3_config(name: str, agent_cfg: SimpleNamespace) -> None:
    alg = agent_cfg.algorithm
    policy = agent_cfg.policy
    print(
        f"[probe step6] {name}: "
        f"experiment_name={agent_cfg.experiment_name} "
        f"is_full_resume={agent_cfg.is_full_resume} "
        f"max_iterations={agent_cfg.max_iterations} "
        f"objective={alg.frontres_training_objective} "
        f"replay_enabled={alg.frontres_segment_replay_enabled} "
        f"live_runner_enabled={alg.frontres_segment_live_runner_enabled} "
        f"live_train_enabled={alg.frontres_segment_live_train_enabled} "
        f"sentinel={alg.frontres_segment_live_sentinel_only} "
        f"probe={alg.frontres_segment_live_probe_only} "
        f"storage={alg.frontres_segment_live_storage_write_only} "
        f"single_update={alg.frontres_segment_live_single_update_only} "
        f"update_loop={alg.frontres_segment_live_update_loop_only} "
        f"update_steps={alg.frontres_segment_live_update_steps} "
        f"hsl_init={alg.frontres_hsl_init_enabled} "
        f"task_corrections={policy.num_task_corrections}",
        flush=True,
    )


def _probe_exception(name: str, exc: Exception) -> None:
    print(f"[probe step6] {name}: exception={type(exc).__name__} message={exc}", flush=True)


def test_stage3_default_enters_live_train_config_without_zeroing_iterations() -> None:
    agent_cfg = _agent_cfg()

    _apply_frontres_stage_preset(
        agent_cfg,
        _args(frontres_segment_live_update_steps=7, frontres_formal_runtime_audit=True),
    )
    _probe_stage3_config("stage3_default_live_train", agent_cfg)

    alg = agent_cfg.algorithm
    assert agent_cfg.experiment_name == "g1_flat_frontres_stage3_segment_hrl"
    assert agent_cfg.is_full_resume is False
    assert agent_cfg.max_iterations == 11
    assert agent_cfg.supervised_warmup_iterations == 0
    assert alg.frontres_training_objective == "segment_replay_hrl"
    assert alg.frontres_segment_replay_enabled is True
    assert alg.frontres_segment_live_runner_enabled is True
    assert alg.frontres_segment_live_train_enabled is True
    assert alg.frontres_segment_live_update_steps == 1
    assert alg.frontres_segment_critic_warmup_iterations == 200
    assert alg.frontres_segment_actor_warmup_iterations == 500
    assert alg.frontres_segment_k_curriculum == (
        (8, 2, 200, 500, 1300, "lower-k8", 0.5, "linear-joint-v1", 1300, 2.381),
        (16, 3, 300, 300, 900, "lower-k16", 0.6, "linear-joint-v1", 900, 2.381),
        (32, 4, 400, 300, 625, "lower-k32", 0.7, "linear-joint-v1", 625, 2.381),
    )
    assert alg.frontres_formal_runtime_audit is True
    assert alg.frontres_hsl_init_enabled is False
    assert alg.frontres_hsl_rollout_label_enabled is False
    assert alg.lambda_supervised == 0.0
    assert alg.lambda_supervised_min == 0.0
    assert alg.frontres_formal_transaction_enabled is True
    assert alg.frontres_future_offsets == (1, 2)
    assert alg.frontres_future_intent_layout_version == "frontres-v015-future-intent-q29-v1"
    assert alg.frontres_segment_k == 8
    assert alg.frontres_segment_max_horizon_k == 64
    assert alg.frontres_segment_advantage_normalization == "grouped_scale_only"
    assert agent_cfg.save_interval == 100
    assert agent_cfg.policy.num_task_corrections == 6
    assert not hasattr(agent_cfg.policy, "task_conf_dim")
    assert not hasattr(agent_cfg.policy, "frontres_split_acceptance_head")

    audit_cfg = _agent_cfg()
    _apply_frontres_stage_preset(
        audit_cfg,
        _args(frontres_checkpoint_interval=1),
    )
    assert audit_cfg.save_interval == 1

    invalid_cfg = _agent_cfg()
    try:
        _apply_frontres_stage_preset(invalid_cfg, _args(frontres_checkpoint_interval=0))
    except ValueError as exc:
        assert "frontres_checkpoint_interval" in str(exc)
    else:
        raise AssertionError("non-positive checkpoint interval must be rejected")

    missing_initializer = _agent_cfg()
    try:
        _apply_frontres_stage_preset(
            missing_initializer,
            _args(frontres_v015_hsl_initializer_checkpoint=None),
        )
    except ValueError as exc:
        assert "frontres_v015_hsl_initializer_checkpoint" in str(exc)
    else:
        raise AssertionError("ordinary Stage-3 training must require an explicit HSL-v2 initializer")

    missing_schedule = _agent_cfg()
    try:
        _apply_frontres_stage_preset(
            missing_schedule,
            _args(frontres_segment_k_curriculum=None),
        )
    except ValueError as exc:
        assert "frontres_segment_k_curriculum" in str(exc)
    else:
        raise AssertionError("ordinary Stage-3 training must require an explicit v009 K curriculum")


def test_stage3_rejects_noncanonical_future_offsets_before_config_mutation() -> None:
    agent_cfg = _agent_cfg()
    before = vars(agent_cfg.algorithm).copy()
    try:
        _apply_frontres_stage_preset(
            agent_cfg,
            _args(frontres_v015_future_offsets="1,3"),
        )
    except ValueError as exc:
        assert "exactly '1,2'" in str(exc)
    else:
        raise AssertionError("Stage-3 must reject noncanonical future offsets")
    assert vars(agent_cfg.algorithm) == before


def test_retired_optimizer_flags_reject_before_stage3_config_mutation() -> None:
    for field in (
        "frontres_segment_live_single_update_only",
        "frontres_segment_live_update_loop_only",
    ):
        agent_cfg = _agent_cfg()
        before = vars(agent_cfg.algorithm).copy()
        try:
            _apply_frontres_stage_preset(agent_cfg, _args(**{field: True}))
        except ValueError as exc:
            assert "FRS-PPO-v005 rejects retired Stage-3" in str(exc)
        else:
            raise AssertionError(f"retired Stage-3 flag must reject: {field}")
        assert vars(agent_cfg.algorithm) == before


def test_stage2_hsl_warmup_constructs_proposal_only_6d_policy() -> None:
    agent_cfg = _agent_cfg()
    try:
        _apply_frontres_stage_preset(
            agent_cfg,
            _args(frontres_stage="stage2_hsl_warmup"),
        )
    except ValueError as exc:
        assert "stage3_segment_hrl" in str(exc)
    else:
        raise AssertionError("v015 rejects the retired standalone Stage-2 HSL route")
def test_default_frontres_policy_config_is_proposal_only_6d() -> None:
    config = MOSAIC_CFG_PATH.read_text(encoding="utf-8")

    assert "num_task_corrections   = 6" in config
    assert "task_conf_dim" not in config
    assert "frontres_split_acceptance_head" not in config
    assert "direct correction proposal = [Δpos(3), Δrpy(3)]" in config


def test_stage3_ppo_schedule_override_is_explicit_parse_arg() -> None:
    agent_cfg = _agent_cfg()

    _apply_frontres_stage_preset(agent_cfg, _args(frontres_segment_ppo_schedule="adaptive"))
    _apply_frontres_segment_ppo_schedule_override(agent_cfg, _args(frontres_segment_ppo_schedule="adaptive"))
    _probe_stage3_config("stage3_ppo_schedule_override", agent_cfg)

    assert agent_cfg.algorithm.schedule == "adaptive"


def test_stage3_ppo_schedule_override_rejects_non_stage3() -> None:
    agent_cfg = _agent_cfg()
    try:
        _apply_frontres_segment_ppo_schedule_override(
            agent_cfg,
            _args(frontres_stage="stage1_hsl", frontres_segment_ppo_schedule="adaptive"),
        )
    except ValueError as exc:
        _probe_exception("rejects_ppo_schedule_without_stage3", exc)
        assert "requires --frontres_stage stage3_segment_hrl" in str(exc)
    else:
        raise AssertionError("PPO schedule override must require Stage 3")


def test_stage3_ppo_lr_override_is_explicit_parse_arg() -> None:
    agent_cfg = _agent_cfg()

    _apply_frontres_stage_preset(agent_cfg, _args(frontres_segment_ppo_lr=1.0e-6))
    _apply_frontres_segment_ppo_lr_override(agent_cfg, _args(frontres_segment_ppo_lr=1.0e-6))
    _probe_stage3_config("stage3_ppo_lr_override", agent_cfg)

    assert agent_cfg.algorithm.learning_rate == 1.0e-6


def test_stage3_ppo_lr_override_rejects_non_stage3() -> None:
    agent_cfg = _agent_cfg()
    try:
        _apply_frontres_segment_ppo_lr_override(
            agent_cfg,
            _args(frontres_stage="stage1_hsl", frontres_segment_ppo_lr=1.0e-6),
        )
    except ValueError as exc:
        _probe_exception("rejects_ppo_lr_without_stage3", exc)
        assert "requires --frontres_stage stage3_segment_hrl" in str(exc)
    else:
        raise AssertionError("PPO LR override must require Stage 3")


def test_stage3_ppo_lr_override_rejects_non_positive_lr() -> None:
    agent_cfg = _agent_cfg()
    try:
        _apply_frontres_segment_ppo_lr_override(
            agent_cfg,
            _args(frontres_segment_ppo_lr=0.0),
        )
    except ValueError as exc:
        _probe_exception("rejects_non_positive_ppo_lr", exc)
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("PPO LR override must reject non-positive values")


def test_stage3_rejects_multiple_live_sentinel_modes() -> None:
    agent_cfg = _agent_cfg()
    try:
        _apply_frontres_stage_preset(
            agent_cfg,
            _args(frontres_segment_live_probe_only=True, frontres_segment_live_storage_write_only=True),
        )
    except ValueError as exc:
        _probe_exception("rejects_multiple_live_sentinel_modes", exc)
        assert "Use only one of" in str(exc)
    else:
        raise AssertionError("Stage 3 must reject multiple live sentinel modes")


def test_live_sentinel_flags_require_stage3() -> None:
    agent_cfg = _agent_cfg()
    try:
        _apply_frontres_stage_preset(
            agent_cfg,
            _args(frontres_stage="stage1_hsl", frontres_segment_live_probe_only=True),
        )
    except ValueError as exc:
        _probe_exception("rejects_live_sentinel_without_stage3", exc)
        assert "stage3_segment_hrl" in str(exc)
    else:
        raise AssertionError("Live sentinel flags must require Stage 3")


def test_stage3_motion_loader_cfg_aligns_with_index_cache() -> None:
    motion_cfg = SimpleNamespace(
        motion_dataset_shard_across_gpus=True,
        motion_dataset_load_cap=512,
        motion_dataset_log_shard_info=False,
    )
    env_cfg = SimpleNamespace(commands=SimpleNamespace(motion=motion_cfg))

    _configure_frontres_stage3_segment_hrl_env_cfg(env_cfg)
    print(
        "[probe bug-index-reset] stage3_motion_loader_cfg: "
        f"load_cap={motion_cfg.motion_dataset_load_cap} "
        f"shard={motion_cfg.motion_dataset_shard_across_gpus} "
        f"log_shard={motion_cfg.motion_dataset_log_shard_info}",
        flush=True,
    )

    assert motion_cfg.motion_dataset_load_cap is None
    assert motion_cfg.motion_dataset_shard_across_gpus is False
    assert motion_cfg.motion_dataset_log_shard_info is True


def test_train_dispatch_orders_stage3_live_path_before_legacy_learn() -> None:
    train = TRAIN_PATH.read_text()
    live_train = "runner.learn_frontres_segment_live("
    legacy_learn = "runner.learn(num_learning_iterations=agent_cfg.max_iterations"
    probe = "runner.run_frontres_segment_live_probe(init_at_random_ep_len=True)"

    print(
        "[probe step6] train_dispatch_order: "
        f"probe_before_legacy={train.index(probe) < train.index(legacy_learn)} "
        f"live_train_before_legacy={train.index(live_train) < train.index(legacy_learn)}",
        flush=True,
    )
    assert train.index(probe) < train.index(legacy_learn)
    assert "runner.run_frontres_segment_live_update_loop(" not in train
    assert "or args_cli.frontres_segment_live_single_update_only" not in train
    assert "run_frontres_segment_sequence_offline_eval" not in train
    assert train.index(live_train) < train.index(legacy_learn)


def test_stage3_hsl_initializer_dispatch_is_explicit_and_formal_training_opens() -> None:
    train = TRAIN_PATH.read_text()
    initializer_call = "runner.load_frontres_hsl_initializer(hsl_initializer)"
    resume_branch = "if agent_cfg.resume:"
    live_train = "runner.learn_frontres_segment_live("
    assert initializer_call in train
    assert train.index(initializer_call) < train.index(resume_branch)
    assert "G3-S1B transaction provider/dispatch" not in train
    assert "refusing the legacy Stage-3 training loop" not in train

    runner_source = (ROOT / "source" / "rsl_rl" / "rsl_rl" / "runners" / "on_policy_runner.py").read_text()
    training_source = (ROOT / "source" / "rsl_rl" / "rsl_rl" / "runners" / "frontres_segment_live_training.py").read_text()
    assert "def load_frontres_hsl_initializer(self, path: str):" in runner_source
    assert "operation = lambda: load_frontres_hsl_initializer(self, path)" in runner_source
    assert 'return self._frontres_startup_lifecycle.load("hsl", operation)' in runner_source
    assert "def resolve_authority():" in runner_source
    assert "resolve_frontres_v015_observation_authority(" in runner_source
    assert "self._frontres_startup_lifecycle.resolve_layout(resolve_authority)" in runner_source
    assert "def run_frontres_formal_training_transaction(" in runner_source
    assert "result = runner.run_frontres_formal_training_transaction(" in training_source
    assert training_source.index("if formal_v015:") < training_source.index("runner.run_frontres_segment_live_update_loop(")
    assert live_train in train
    print(
        "[T-HSL-explicit/T-formal-dispatch/T-legacy-isolation] initializer precedes resume and ordinary Stage-3 selects the formal owner",
        flush=True,
    )


if __name__ == "__main__":
    test_stage3_default_enters_live_train_config_without_zeroing_iterations()
    test_stage3_rejects_noncanonical_future_offsets_before_config_mutation()
    test_retired_optimizer_flags_reject_before_stage3_config_mutation()
    test_stage2_hsl_warmup_constructs_proposal_only_6d_policy()
    test_default_frontres_policy_config_is_proposal_only_6d()
    test_stage3_ppo_schedule_override_is_explicit_parse_arg()
    test_stage3_ppo_schedule_override_rejects_non_stage3()
    test_stage3_ppo_lr_override_is_explicit_parse_arg()
    test_stage3_ppo_lr_override_rejects_non_stage3()
    test_stage3_ppo_lr_override_rejects_non_positive_lr()
    test_stage3_rejects_multiple_live_sentinel_modes()
    test_live_sentinel_flags_require_stage3()
    test_stage3_motion_loader_cfg_aligns_with_index_cache()
    test_train_dispatch_orders_stage3_live_path_before_legacy_learn()
    test_stage3_hsl_initializer_dispatch_is_explicit_and_formal_training_opens()
    print("frontres_segment_stage3_entrypoint_pseudo_contract: ok")
