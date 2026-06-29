#!/usr/bin/env python3
"""FrontRES stage entrypoint contract.

Stage 1/2 are live.  Stage 3 is recognized as Segment Replay HRL.  Explicit
sentinel/probe/storage/single-update/update-loop sentinels and the dedicated
live train loop can enter the minimal runner path.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def _between(text: str, start: str, end: str) -> str:
    start_i = text.index(start)
    end_i = text.index(end, start_i)
    return text[start_i:end_i]


def main() -> None:
    train = _read("scripts/rsl_rl/train.py")
    stage1_cache = _read("run/run_frontres_stage1_segment_cache.sh")
    stage1 = _read("run/run_frontres_stage1_hsl.sh")
    stage2 = _read("run/run_frontres_stage2_acceptance.sh")
    stage3 = _read("run/run_frontres_stage3_segment_hrl.sh")

    assert 'choices=("stage1_segment_cache", "stage1_hsl", "stage2_hsl_warmup", "stage2_acceptance", "stage3_segment_hrl")' in train
    assert '"--frontres_segment_cache_dir"' in train
    assert '"--frontres_segment_cache_k"' in train
    assert '"--frontres_segment_cache_frame_stride"' in train
    assert '"--frontres_segment_cache_max_motions"' in train
    assert '"--frontres_segment_cache_max_segments"' in train
    assert '"--frontres_segment_cache_variants_per_strength"' in train
    assert '"--frontres_segment_cache_perturbation_strengths"' in train
    assert '"--frontres_segment_live_sentinel_only"' in train
    assert '"--frontres_segment_live_probe_only"' in train
    assert '"--frontres_segment_live_storage_write_only"' in train
    assert '"--frontres_segment_live_single_update_only"' in train
    assert '"--frontres_segment_live_update_loop_only"' in train
    assert '"--frontres_segment_live_update_steps"' in train
    assert "Stage 3 live sentinel/probe/storage/update flags require --frontres_stage stage3_segment_hrl" in train
    assert 'if stage == "stage1_segment_cache":' in train
    assert 'elif stage in ("stage1_hsl", "stage2_hsl_warmup"):' in train
    assert 'elif stage == "stage2_acceptance":' in train
    assert 'elif stage == "stage3_segment_hrl":' in train
    assert 'FEMR_LOG_ROOT' in train
    assert 'os.path.dirname(__file__)' in train
    assert 'def _prefer_local_femr_sources()' in train
    assert '_prefer_local_femr_sources()' in train
    assert '"source/rsl_rl"' in train
    assert '"source/whole_body_tracking"' in train
    assert 'candidate_base_paths' not in train
    assert '"/workspace/"' not in train
    assert '"/hdd1/cyx/MOSAIC/"' not in train
    assert train.index("_prefer_local_femr_sources()") < train.index("from isaaclab.app import AppLauncher")
    assert train.index("_prefer_local_femr_sources()") < train.index("from whole_body_tracking.utils.my_on_policy_runner")
    assert 'stage2_authority' not in _between(
        train,
        '''parser.add_argument(
    "--frontres_stage"''',
        '''parser.add_argument(
    "--supervised_warmup_iterations"''',
    )

    stage1_cache_block = _between(train, 'if stage == "stage1_segment_cache":', '    elif stage in ("stage1_hsl", "stage2_hsl_warmup"):')
    stage1_cache_required = [
        'agent_cfg.experiment_name = "g1_flat_frontres_stage1_segment_cache"',
        'agent_cfg.max_iterations = 0',
        '_set_if_present(alg_cfg, "frontres_segment_replay_enabled", True)',
        '_set_if_present(alg_cfg, "frontres_segment_k", max(1, int(getattr(args_cli, "frontres_segment_cache_k", 4))))',
        '_set_if_present(alg_cfg, "frontres_segment_live_runner_enabled", False)',
    ]
    for needle in stage1_cache_required:
        assert needle in stage1_cache_block, needle
    assert "def _configure_frontres_stage1_segment_cache_env_cfg" in train
    assert "def _run_frontres_stage1_segment_cache(env, args_cli, log_dir: str)" in train
    assert 'return "/hdd1/cyx/AMASS_G1Segment"' in train
    assert "[FrontRES Stage1 Segment Cache] live_sentinel" in train
    assert "FrontRESStage1EnvAdapter" in train
    assert "build_stage1_segment_cache" in train
    assert "Stage 1 Segment Cache entrypoint is recognized" not in train
    assert "NotImplementedError" not in _between(
        train,
        "def _run_frontres_stage1_segment_cache(env, args_cli, log_dir: str)",
        "@hydra_task_config",
    )
    assert 'if args_cli.frontres_stage == "stage1_segment_cache":' in train
    assert train.index("_configure_frontres_stage1_segment_cache_env_cfg(env_cfg, args_cli)") < train.index("gym.make(")
    assert train.index("gym.make(") < train.index("_run_frontres_stage1_segment_cache(env, args_cli, log_dir)")
    assert train.index("_run_frontres_stage1_segment_cache(env, args_cli, log_dir)") < train.index("RslRlVecEnvWrapper(env)")

    stage2_block = _between(train, 'elif stage == "stage2_acceptance":', '    print(f"[FrontRES Stage]')
    required = [
        'agent_cfg.experiment_name = "g1_flat_frontres_stage2_acceptance"',
        'agent_cfg.is_full_resume = False',
        'agent_cfg.supervised_warmup_iterations = 0',
        '_set_if_present(alg_cfg, "frontres_training_objective", "hsl_hybrid")',
        '_set_if_present(alg_cfg, "frontres_acceptance_preference_weight", 1.0)',
        '_set_if_present(policy_cfg, "frontres_split_acceptance_head", True)',
        '_set_if_present(alg_cfg, "frontres_authority_actor_critic_enabled", False)',
        '_set_if_present(alg_cfg, "frontres_authority_actor_loss_weight", 0.0)',
        '_set_if_present(alg_cfg, "frontres_authority_critic_loss_weight", 0.0)',
        '_set_if_present(alg_cfg, "frontres_structured_joint_rl_enabled", False)',
        '_set_if_present(alg_cfg, "frontres_structured_joint_rl_weight", 0.0)',
        '_set_if_present(alg_cfg, "frontres_structured_joint_prior_loss_weight", 0.0)',
        '_set_if_present(policy_cfg, "frontres_authority_actor_critic", False)',
        '_set_if_present(policy_cfg, "frontres_state_router_enabled", False)',
    ]
    for needle in required:
        assert needle in stage2_block, needle
    forbidden = [
        'frontres_authority_actor_critic_enabled", True',
        'frontres_authority_actor_loss_weight", 1.0',
        'frontres_authority_critic_loss_weight", 1.0',
        'frontres_structured_joint_enabled"',
        'frontres_structured_joint_prior_weight"',
        'frontres_structured_joint_rl_enabled", True',
        'frontres_structured_joint_rl_weight", 1.0',
        'frontres_authority_return_horizon", 8',
        'frontres_perturbation_temporal_mode", "burst"',
    ]
    for needle in forbidden:
        assert needle not in stage2_block, needle

    stage3_block = _between(train, 'elif stage == "stage3_segment_hrl":', '    print(f"[FrontRES Stage]')
    stage3_required = [
        'agent_cfg.experiment_name = "g1_flat_frontres_stage3_segment_hrl"',
        'agent_cfg.is_full_resume = False',
        'agent_cfg.supervised_warmup_iterations = 0',
        '_set_if_present(alg_cfg, "frontres_training_objective", "segment_replay_hrl")',
        '_set_if_present(alg_cfg, "frontres_segment_replay_enabled", True)',
        'live_sentinel_only = live_sentinel_arg',
        'live_probe_only = live_probe_arg',
        'live_storage_only = live_storage_arg',
        'live_single_update_only = live_single_update_arg',
        'live_update_loop_only = live_update_loop_arg',
        'live_update_steps = max(1, int(getattr(args_cli, "frontres_segment_live_update_steps", 4)))',
        'live_train_enabled = not (',
        'Use only one of --frontres_segment_live_sentinel_only',
        'agent_cfg.max_iterations = 0',
        '"frontres_segment_live_runner_enabled",',
        'or live_train_enabled',
        '_set_if_present(alg_cfg, "frontres_segment_live_sentinel_only", live_sentinel_only)',
        '_set_if_present(alg_cfg, "frontres_segment_live_probe_only", live_probe_only)',
        '_set_if_present(alg_cfg, "frontres_segment_live_storage_write_only", live_storage_only)',
        '_set_if_present(alg_cfg, "frontres_segment_live_single_update_only", live_single_update_only)',
        '_set_if_present(alg_cfg, "frontres_segment_live_update_loop_only", live_update_loop_only)',
        '_set_if_present(alg_cfg, "frontres_segment_live_train_enabled", live_train_enabled)',
        '_set_if_present(alg_cfg, "frontres_segment_live_update_steps", live_update_steps)',
        '_set_if_present(alg_cfg, "frontres_hsl_init_enabled", True)',
        '_set_if_present(alg_cfg, "frontres_segment_k", 4)',
        '_set_if_present(alg_cfg, "frontres_segment_sampler_global_frac", 0.4)',
        '_set_if_present(alg_cfg, "frontres_segment_sampler_replay_frac", 0.5)',
        '_set_if_present(alg_cfg, "frontres_segment_sampler_review_frac", 0.1)',
        '_set_if_present(alg_cfg, "frontres_segment_reset_mode", "auto")',
        '_set_if_present(alg_cfg, "frontres_acceptance_preference_weight", 0.0)',
        '_set_if_present(policy_cfg, "frontres_split_acceptance_head", False)',
        '_set_if_present(alg_cfg, "frontres_authority_actor_critic_enabled", False)',
        '_set_if_present(alg_cfg, "frontres_structured_joint_rl_enabled", False)',
    ]
    for needle in stage3_required:
        assert needle in stage3_block, needle
    stage3_forbidden = [
        'frontres_acceptance_preference_weight", 1.0',
        'frontres_split_acceptance_head", True',
        'frontres_authority_actor_critic_enabled", True',
        'frontres_structured_joint_rl_enabled", True',
    ]
    for needle in stage3_forbidden:
        assert needle not in stage3_block, needle

    algorithm_cfg = _read("source/rsl_rl/rsl_rl/modules/rsl_rl_cfg.py")
    task_cfg = _read("source/whole_body_tracking/whole_body_tracking/utils/rsl_rl_cfg.py")
    algorithm_impl = _read("source/rsl_rl/rsl_rl/algorithms/frontres_unified.py")
    runner_impl = _read("source/rsl_rl/rsl_rl/runners/on_policy_runner.py")
    live_probe_helper = _read("source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py")
    live_training_helper = _read("source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py")
    for cfg_text in (algorithm_cfg, task_cfg):
        assert "frontres_segment_replay_enabled: bool = False" in cfg_text
        assert "frontres_segment_live_runner_enabled: bool = False" in cfg_text
        assert "frontres_segment_live_sentinel_only: bool = False" in cfg_text
        assert "frontres_segment_live_probe_only: bool = False" in cfg_text
        assert "frontres_segment_live_storage_write_only: bool = False" in cfg_text
        assert "frontres_segment_live_single_update_only: bool = False" in cfg_text
        assert "frontres_segment_live_update_loop_only: bool = False" in cfg_text
        assert "frontres_segment_live_train_enabled: bool = False" in cfg_text
        assert "frontres_segment_live_update_steps: int = 4" in cfg_text
        assert "frontres_segment_live_fail_on_invalid_update: bool = True" in cfg_text
        assert "frontres_segment_live_min_valid_count: int = 1" in cfg_text
        assert "frontres_segment_live_fail_on_nonfinite: bool = True" in cfg_text
        assert "frontres_hsl_init_enabled: bool = False" in cfg_text
        assert "frontres_segment_k: int = 4" in cfg_text
        assert "frontres_segment_sampler_global_frac: float = 0.4" in cfg_text
        assert "frontres_segment_sampler_replay_frac: float = 0.5" in cfg_text
        assert "frontres_segment_sampler_review_frac: float = 0.1" in cfg_text
        assert 'frontres_segment_reset_mode: str = "auto"' in cfg_text
    assert 'frontres_training_objective == "segment_replay_hrl"' in algorithm_impl
    assert "live runner integration is disabled" in algorithm_impl
    assert "runner/PPO integration is not wired yet" in algorithm_impl
    assert "frontres_segment_live_sentinel_only" in algorithm_impl
    assert "frontres_segment_live_probe_only" in algorithm_impl
    assert "frontres_segment_live_storage_write_only" in algorithm_impl
    assert "frontres_segment_live_single_update_only" in algorithm_impl
    assert "frontres_segment_live_update_loop_only" in algorithm_impl
    assert "frontres_segment_live_train_enabled" in algorithm_impl
    assert "frontres_segment_live_update_steps" in algorithm_impl
    assert "frontres_segment_live_fail_on_invalid_update" in algorithm_impl
    assert "frontres_segment_live_min_valid_count" in algorithm_impl
    assert "frontres_segment_live_fail_on_nonfinite" in algorithm_impl
    assert "runner will execute exactly one PPO optimizer step and exit" in algorithm_impl
    assert "PPO optimizer steps and exit" in algorithm_impl
    assert "PPO optimizer steps per iteration" in algorithm_impl
    assert "args_cli.frontres_segment_live_single_update_only" in train
    assert "args_cli.frontres_segment_live_update_loop_only" in train
    assert "runner.run_frontres_segment_live_update_loop(init_at_random_ep_len=True)" in train
    assert "runner.learn_frontres_segment_live(" in train
    assert "frontres_segment_live_train_enabled" in train
    assert "runner.run_frontres_segment_live_probe(init_at_random_ep_len=True)" in train
    assert "run_frontres_segment_live_probe_helper(self" in runner_impl
    assert "FrontRESSegmentRolloutStorage" not in runner_impl
    assert "FrontRESSegmentTransition" not in runner_impl
    assert "compute_frontres_segment_ppo_loss" not in runner_impl
    assert "FrontRESSegmentPPOConfig" not in runner_impl
    assert "def run_frontres_segment_live_probe" in live_probe_helper
    assert "def build_live_segment_storage" in live_probe_helper
    assert "def run_frontres_segment_single_update" in live_probe_helper
    assert "def _run_live_rollout_capture" in live_probe_helper
    assert "FrontRESSegmentRolloutStorage" in live_probe_helper
    assert "FrontRESSegmentTransition" in live_probe_helper
    assert "compute_frontres_segment_ppo_loss" in live_probe_helper
    assert "FrontRESSegmentPPOConfig" in live_probe_helper
    assert "run_frontres_segment_live_training_loop" in runner_impl
    assert "FrontRES Segment live update produced update_count=0" in live_training_helper
    assert "FrontRES Segment live update produced non-finite" in live_training_helper
    assert "too few valid PPO samples" in live_training_helper
    assert "def run_frontres_segment_live_training_loop" in live_training_helper
    assert "FrontRES Segment live update summary missing keys" in live_training_helper
    assert train.index("runner.run_frontres_segment_live_probe(init_at_random_ep_len=True)") < train.index(
        "runner.learn(num_learning_iterations=agent_cfg.max_iterations"
    )

    assert '--frontres_stage stage1_segment_cache' in stage1_cache
    assert '--frontres_segment_cache_k "${SEGMENT_K}"' in stage1_cache
    assert '--frontres_segment_cache_frame_stride "${FRAME_STRIDE}"' in stage1_cache
    assert '--frontres_segment_cache_max_motions "${MAX_MOTIONS}"' in stage1_cache
    assert '--frontres_segment_cache_max_segments "${MAX_SEGMENTS}"' in stage1_cache
    assert '--frontres_segment_cache_variants_per_strength "${VARIANTS_PER_STRENGTH}"' in stage1_cache
    assert '--frontres_segment_cache_perturbation_strengths "${PERTURBATION_STRENGTHS}"' in stage1_cache
    assert 'CACHE_DIR="${4:-/hdd1/cyx/AMASS_G1Segment}"' in stage1_cache
    assert 'g1_flat_frontres_stage1_segment_cache' in stage1_cache
    assert 'Stage 1 builds the Segment Replay cache' in stage1_cache
    assert '--frontres_stage stage1_hsl' in stage1
    assert 'authority' not in stage1.lower()
    assert '--frontres_stage stage2_acceptance' in stage2
    assert '--resume_student_checkpoint "${STAGE1_CHECKPOINT}"' in stage2
    assert '--is_full_resume False' in stage2
    assert 'g1_flat_frontres_stage2_acceptance' in stage2
    assert 'stage2_authority' not in stage2
    assert '--frontres_stage stage3_segment_hrl' in stage3
    assert '--resume_student_checkpoint "${STAGE1_CHECKPOINT}"' in stage3
    assert '--is_full_resume False' in stage3
    assert '--frontres_segment_live_update_steps "${UPDATE_STEPS}"' in stage3
    assert '--frontres_segment_live_update_loop_only' in stage3
    assert 'FRONTRES_STAGE3_RUN_CONTRACTS' in stage3
    assert 'frontres_segment_all_contract_suite.py' in stage3
    assert '[FrontRES Stage3 contract preflight] PASS' in stage3
    assert 'g1_flat_frontres_stage3_segment_hrl' in stage3
    assert 'stage2_acceptance' not in stage3
    assert '"/hdd1/cyx/MOSAIC/"' not in stage3
    assert not (ROOT / 'run/run_frontres_stage2_authority.sh').exists()
    legacy = ROOT / 'run/legacy/run_frontres_stage2_authority.sh'
    assert legacy.exists()
    assert 'Legacy ablation entrypoint' in legacy.read_text()
    print("PASS: FrontRES Stage 1/2 live presets and Stage 3 Segment Replay contract are explicit.")


if __name__ == "__main__":
    main()
