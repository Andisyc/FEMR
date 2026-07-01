#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[4]
TRAIN_PATH = ROOT / "scripts" / "rsl_rl" / "train.py"


def _load_stage_preset():
    tree = ast.parse(TRAIN_PATH.read_text())
    wanted = {"_set_if_present", "_apply_frontres_stage_preset", "_configure_frontres_stage3_segment_hrl_env_cfg"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"RslRlOnPolicyRunnerCfg": object}
    exec(compile(module, str(TRAIN_PATH), "exec"), namespace)
    return namespace["_apply_frontres_stage_preset"], namespace["_configure_frontres_stage3_segment_hrl_env_cfg"]


_apply_frontres_stage_preset, _configure_frontres_stage3_segment_hrl_env_cfg = _load_stage_preset()


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
        frontres_segment_live_update_steps=4,
        frontres_hsl_init_enabled=False,
        frontres_segment_k=0,
        frontres_segment_sampler_global_frac=0.0,
        frontres_segment_sampler_replay_frac=0.0,
        frontres_segment_sampler_review_frac=0.0,
        frontres_segment_reset_mode="unset",
        frontres_acceptance_preference_weight=1.0,
        frontres_state_alpha_weight=1.0,
        frontres_authority_actor_critic_enabled=True,
        frontres_authority_actor_loss_weight=1.0,
        frontres_authority_critic_loss_weight=1.0,
        frontres_structured_joint_rl_enabled=True,
        frontres_structured_joint_rl_weight=1.0,
        frontres_structured_joint_prior_loss_weight=1.0,
    )


def _policy_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        frontres_split_acceptance_head=True,
        frontres_authority_actor_critic=True,
        frontres_state_router_enabled=True,
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
        ppo_actor_warmup_iterations=99,
        ppo_actor_ramp_iterations=99,
        max_iterations=11,
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
        "experiment_name": None,
        "is_full_resume": None,
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
        f"acceptance_weight={alg.frontres_acceptance_preference_weight} "
        f"split_acceptance_head={policy.frontres_split_acceptance_head}",
        flush=True,
    )


def _probe_exception(name: str, exc: Exception) -> None:
    print(f"[probe step6] {name}: exception={type(exc).__name__} message={exc}", flush=True)


def test_stage3_default_enters_live_train_config_without_zeroing_iterations() -> None:
    agent_cfg = _agent_cfg()

    _apply_frontres_stage_preset(agent_cfg, _args(frontres_segment_live_update_steps=7))
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
    assert alg.frontres_segment_live_update_steps == 7
    assert alg.frontres_hsl_init_enabled is True
    assert alg.frontres_acceptance_preference_weight == 0.0
    assert agent_cfg.policy.frontres_split_acceptance_head is False


def test_stage3_sentinel_zeroes_iterations_and_disables_live_train() -> None:
    agent_cfg = _agent_cfg()

    _apply_frontres_stage_preset(
        agent_cfg,
        _args(frontres_segment_live_single_update_only=True, frontres_segment_live_update_steps=3),
    )
    _probe_stage3_config("stage3_single_update_sentinel", agent_cfg)

    alg = agent_cfg.algorithm
    assert agent_cfg.max_iterations == 0
    assert alg.frontres_segment_live_runner_enabled is True
    assert alg.frontres_segment_live_train_enabled is False
    assert alg.frontres_segment_live_single_update_only is True
    assert alg.frontres_segment_live_update_steps == 3


def test_stage3_rejects_multiple_live_sentinel_modes() -> None:
    agent_cfg = _agent_cfg()
    try:
        _apply_frontres_stage_preset(
            agent_cfg,
            _args(frontres_segment_live_probe_only=True, frontres_segment_live_update_loop_only=True),
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
            _args(frontres_stage="stage2_acceptance", frontres_segment_live_probe_only=True),
        )
    except ValueError as exc:
        _probe_exception("rejects_live_sentinel_without_stage3", exc)
        assert "require --frontres_stage stage3_segment_hrl" in str(exc)
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
    update_loop = "runner.run_frontres_segment_live_update_loop(init_at_random_ep_len=True)"
    probe = "runner.run_frontres_segment_live_probe(init_at_random_ep_len=True)"

    print(
        "[probe step6] train_dispatch_order: "
        f"probe_before_legacy={train.index(probe) < train.index(legacy_learn)} "
        f"update_loop_before_legacy={train.index(update_loop) < train.index(legacy_learn)} "
        f"live_train_before_legacy={train.index(live_train) < train.index(legacy_learn)}",
        flush=True,
    )
    assert train.index(probe) < train.index(legacy_learn)
    assert train.index(update_loop) < train.index(legacy_learn)
    assert train.index(live_train) < train.index(legacy_learn)


if __name__ == "__main__":
    test_stage3_default_enters_live_train_config_without_zeroing_iterations()
    test_stage3_sentinel_zeroes_iterations_and_disables_live_train()
    test_stage3_rejects_multiple_live_sentinel_modes()
    test_live_sentinel_flags_require_stage3()
    test_stage3_motion_loader_cfg_aligns_with_index_cache()
    test_train_dispatch_orders_stage3_live_path_before_legacy_learn()
    print("frontres_segment_stage3_entrypoint_pseudo_contract: ok")
